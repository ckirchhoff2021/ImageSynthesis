import albumentations as albu


def get_transforms(size: int, scope: str = 'geometric', crop='random'):
    augs = {'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(always_apply=True),
                                     albu.ShiftScaleRotate(always_apply=True),
                                     albu.Transpose(always_apply=True),
                                     albu.OpticalDistortion(always_apply=True),
                                     albu.ElasticTransform(always_apply=True),
                                     ])
            }

    aug_fn = augs[scope]
    crop_fn = {'random': albu.RandomCrop(size, size, always_apply=True),
               'center': albu.CenterCrop(size, size, always_apply=True)}[crop]
    pad = albu.PadIfNeeded(size, size)

    pipeline = albu.Compose([aug_fn, pad, crop_fn], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process


def _resolve_aug_fn(name):
    d = {
        'cutout': albu.Cutout,
        'rgb_shift': albu.RGBShift,
        'hsv_shift': albu.HueSaturationValue,
        'motion_blur': albu.MotionBlur,
        'median_blur': albu.MedianBlur,
        'snow': albu.RandomSnow,
        'shadow': albu.RandomShadow,
        'fog': albu.RandomFog,
        'brightness_contrast': albu.RandomBrightnessContrast,
        'gamma': albu.RandomGamma,
        'sun_flare': albu.RandomSunFlare,
        'sharpen': albu.Sharpen,
        'jpeg': albu.ImageCompression,
        'gray': albu.ToGray,
        'pixelize': albu.Downscale,
        # ToDo: partial gray
    }
    return d[name]


def get_corrupt_function():
    augs = []
    params = dict()
    prob = 0.5
    params['cutout'] = {
        'num_holes': 3,
        'max_h_size': 25,
        'max_w_size': 25
    }
    params['jpeg'] = {
        'quality_lower': 70,
        'quality_upper': 90
    }

    names = ['cutout', 'jpeg', 'motion_blur', 'median_blur', 'gamma', 'rgb_shift', 'hsv_shift', 'sharpen']
    for name in names:
        cls = _resolve_aug_fn(name)
        if name in params:
            augs.append(cls(p=prob, **params[name]))
        else:
            augs.append(cls(p=prob))

    augs = albu.OneOf(augs)
    def process(x):
        return augs(image=x)['image']
    return process
