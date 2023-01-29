from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch


def get_diffusion_model_from_args(args):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )

    diffusion = GaussianDiffusionCustom(
        model,
        image_size=(args.img_height, args.img_width),
    )
    return diffusion


# the out-the-box diffusion only produces square images, but its pretty simple to adjust that
class GaussianDiffusionCustom(GaussianDiffusion):
    def __init__(self,
                 model,
                 *,
                 image_size):
        GaussianDiffusion.__init__(self, model = model, image_size = image_size)

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size[0], image_size[1]),
                         return_all_timesteps=return_all_timesteps)

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
