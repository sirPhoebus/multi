from transformers import AutoProcessor, AutoModelForConditionalGeneration
from PIL import Image
import requests
import torch

# Load model and processor
model_id = "zai-org/GLM-4.5V-FP8"
model = AutoModelForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Example image loading (replace with your image path or URL)
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Prepare the prompt
prompt = "Describe this car in detail."
messages = [
    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
]

# Apply chat template and preprocess image
input_ids = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
pixel_values = processor.preprocess_images(image, return_tensors="pt")

# Generate response
with torch.no_grad():
    output_ids = model.generate(
        input_ids.to(model.device),
        pixel_values=pixel_values.to(model.device),
        max_new_tokens=512
    )

response = processor.decode(output_ids[0], skip_special_tokens=True)
print(response)
