from .voc import VOC, VOCAug, VOCAugBox


def get_dataset(name):
    return {
        "voc": VOC,
        "vocaug": VOCAug,
        "vocaugbox": VOCAugBox,
    }[name]
