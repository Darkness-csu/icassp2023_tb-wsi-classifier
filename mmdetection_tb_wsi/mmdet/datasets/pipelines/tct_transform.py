import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class SelfNormalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, to_rgb=True, debug=False):
        self.to_rgb = to_rgb
        self.debug = debug

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy().astype(np.float32)
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
            if self.to_rgb:
                mean = mean[::-1]
                std = std[::-1]
            results[key] = mmcv.imnormalize(results[key], mean, std, self.to_rgb)

            if self.debug:
                print(img.shape)
                print(mean, std)
                print(results[key])
        results['img_norm_cfg'] = dict(mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
