# --- configs ---
thing_classes = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis"
]
category_name_to_id = {class_name: index for index, class_name in enumerate(thing_classes)}

def get_thing_classes():
    return thing_classes

def get_category_name_to_id():
    thing_classes = get_thing_classes()
    category_name_to_id = {class_name: index for index, class_name in enumerate(thing_classes)}

    return category_name_to_id

def get_class_dict(include_no_finding=False):
    thing_classes = get_thing_classes()
    if include_no_finding:
        thing_classes.append("No finding")
        
    class_dict = {index : class_name for index, class_name in enumerate(thing_classes)}

    return class_dict