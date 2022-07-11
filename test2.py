# Fixed parameters
icon_path = "icon.png"
model_name = "vqgan_imagenet_f16_16384"
seed = 42

texts = "Wild ocean and storm"  # @param {type:"string"}
width = 1000  # @param {type:"integer"}
height = 1000  # @param {type:"integer"}
init_image = ""  # @param {type:"string"}
init_image_icon = False  # @param {type:"boolean"}
if init_image_icon:
    assert os.path.exists(
        icon_path
    ), "No icon has been generated from the previous cell"
    init_image = icon_path

target_images = ""  # @param {type:"string"}
target_image_icon = False  # @param {type:"boolean"}
if target_image_icon:
    assert os.path.exists(
        icon_path
    ), "No icon has been generated from the previous cell"
    target_images = icon_path

# @markdown ---
learning_rate = 0.2  # @param {type:"slider", min:0.00, max:0.30, step:0.01}
max_steps = 200  # @param {type:"integer"}
# images_interval = 50 #@param {type:"integer"}
images_interval = 50  # @param {type:"integer"}

gen_config = {
    "texts": texts,
    "width": width,
    "height": height,
    "init_image": "<icon>" if init_image_icon else init_image,
    "target_images": "<icon>" if target_image_icon else target_images,
    "learning_rate": learning_rate,
    "max_steps": max_steps,
    "training_seed": 42,
    "model": "vqgan_imagenet_f16_16384",
}
