import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import ResidualBlockWithStride, \
    ResidualBlockUpsample

from models.swin_transformer.modules import ConvTransBlock, SWAtten
from models.swin_transformer.basic_blocks import subpel_conv3x3
from utils.helper_functions import update_registered_buffer, get_scale_table, ste_round


class TCM(CompressionModel):
    def __init__(self,
                 config=[2, 2, 2, 2, 2, 2],
                 head_dim=[8, 16, 32, 32, 16, 8],
                 drop_path_rate=0,
                 N=128,
                 M=320,
                 num_slices=5,
                 max_support_slices=5,
                 **kwargs
                 ):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices,
        dim = N,
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [
            ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i + begin],
                           type="W" if not i % 2 else "SW")
            for i in range(config[0])
        ]
        self.m_down1 += [ResidualBlockWithStride(2 * N, 2 * N, stride=2)]

        begin += config[0]
        self.m_down2 = [
            ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i + begin],
                           type="W" if not i % 2 else "SW")
            for i in range(config[1])
        ]
        self.m_down2 += [ResidualBlockWithStride(2 * N, 2 * N, stride=2)]

        begin += config[1]
        self.m_down3 = [
            ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i + begin],
                           type="W" if not i % 2 else "SW")
            for i in range(config[2])
        ]
        self.m_down3 += [nn.Conv2d(2 * N, M, kernel_size=3, stride=2, padding=1)]

        begin += config[2]
        self.m_up1 = [
            ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i + begin],
                           type="W" if not i % 2 else "SW")
            for i in range(config[3])
        ]
        self.m_up1 += [ResidualBlockUpsample(2 * N, 2 * N, 2)]

        self.m_up2 = [
            ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i + begin],
                           type="W" if not i % 2 else "SW")
            for i in range(config[4])
        ]
        self.m_up2 += [ResidualBlockUpsample(2 * N, 2 * N, 2)]

        self.m_up3 = [
            ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i + begin],
                           type="W" if not i % 2 else "SW")
            for i in range(config[5])
        ]
        self.m_up3 += [ResidualBlockUpsample(2 * N, 3, 2)]

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, 2 * N, 2),
            *self.m_down1,
            *self.m_down2,
            *self.m_down3
        )
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M, 2 * N, 2),
            *self.m_up1,
            *self.m_up2,
            *self.m_up3
        )

        self.ha_down1 = [
            ConvTransBlock(N, N, 32, 4, 0, type="W" if not i % 2 else "SW")
            for i in range(config[0])
        ]
        self.ha_down1 += [nn.Conv2d(2 * N, 192, kernel_size=3, stride=2, padding=1)]

        self.h_a = nn.Sequential(
            ResidualBlockWithStride(320, 2 * N, 2),
            *self.ha_down1
        )

        self.hs_up1 = [
            ConvTransBlock(conv_dim=N, trans_dim=N, head_dim=32, window_size=4, drop_path=0,
                           type="W" if not i % 2 else "SW")
            for i in range(config[0])
        ]
        self.hs_up1.append(subpel_conv3x3(2 * N, 32, 2))

        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample(192, 2 * N, 2),
            *self.hs_up1
        )

        self.hs_up2 = [
            ConvTransBlock(N, N, 32, 4, 0, type="W" if not i % 2 else "SW")
            for i in range(config[3])
        ]
        self.hs_up2.append(subpel_conv3x3(2 * N, 320, 2))

        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample(192, 2 * N, 2),
            *self.hs_up2
        )

        # Attention mean
        atten_mean_modules = [SWAtten(
            input_dim=(320, 320 // self.num_slices) * min(i, 5),
            output_dim=(320 + 320 // self.num_slices * min(i, 5)),
            head_dim=16,
            window_size=self.window_size,
            drop_path=0,
            inter_dim=128
        ) for i in range(self.num_slices)
        ]

        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                *atten_mean_modules
            )
        )

        # Attention scale
        atten_scale_modules = [
            [SWAtten(
                input_dim=(320, 320 // self.num_slices) * min(i, 5),
                output_dim=(320 + 320 // self.num_slices * min(i, 5)),
                head_dim=16,
                window_size=self.window_size,
                drop_path=0,
                inter_dim=128
            ) for i in range(self.num_slices)
            ]
        ]
        self.atten_scale = nn.ModuleList(
            nn.Sequential(*atten_scale_modules)
        )

        #
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=(320 + 320 // self.num_slices * min(i, 5)),
                          out_channels=224,
                          stride=1,
                          kernel_size=3,
                          padding=3 // 2
                          ),
                nn.GELU(),
                nn.Conv2d(224, 128, stride=1, kernel_size=3, padding=3 // 2),
                nn.GELU(),
                nn.Conv2d(in_channels=128,
                          out_channels=(320 + 320 // self.num_slices * min(i, 5)),
                          stride=1,
                          kernel_size=3,
                          padding=3 // 2
                          )
            ) for i in range(self.num_slices)
        )

        #
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=(320 + 320 // self.num_slices * min(i, 5)),
                          out_channels=224,
                          stride=1,
                          kernel_size=3,
                          padding=3 // 2
                          ),
                nn.GELU(),
                nn.Conv2d(224, 128, stride=1, kernel_size=3, padding=3 // 2),
                nn.GELU(),
                nn.Conv2d(in_channels=128,
                          out_channels=(320 + 320 // self.num_slices * min(i, 5)),
                          stride=1,
                          kernel_size=3,
                          padding=3 // 2
                          )
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=(320 + 320 // self.num_slices * min(i, 5)),
                          out_channels=224,
                          stride=1,
                          kernel_size=3,
                          padding=3 // 2
                          ),
                nn.GELU(),
                nn.Conv2d(224, 128, stride=1, kernel_size=3, padding=3 // 2),
                nn.GELU(),
                nn.Conv2d(in_channels=128,
                          out_channels=(320 + 320 // self.num_slices * min(i, 5)),
                          stride=1,
                          kernel_size=3,
                          padding=3 // 2
                          )
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def load_state_dict(self, state_dict):
        update_registered_buffer(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict=state_dict
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)

        # update could be a set type so the | operator means union
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        """
        input x with the shape of (B, 3, H, W)
        """
        y = self.g_a(x)  # (B, M=320, H, W)
        y_shape = y.shape[2:]
        z = self.h_a(y)  # (B, 192, H, W)

        # Important DUCANH
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, dim=1)

        y_hat_slices = []
        y_likelihoods = []
        mu_list = []
        scale_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            # mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, y_shape[0], y_shape[1]]
            mu_list.append(mu)

            # Scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, y_shape[0], y_shape[1]]
            scale_list.append(scale)

            # y_slice
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihoods.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            # lrp
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihood": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y}
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, dim=1)

        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            # mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, y_shape[0], y_shape[1]]

            # Scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, y_shape[0], y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "string": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()   

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            # mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, y_shape[0], y_shape[1]]

            # Scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, y_shape[0], y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices)
        x_hat = self.g_s(y_hat).clamp(0, 1)     # ?? Clamp DUCANH

        return {
            "x_hat": x_hat
        }
