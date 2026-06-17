from utils.inference import run_inference_and_save

result = run_inference_and_save(
    image_path="static/uploads/test_side.jpg",
    overlay_save_path="static/results/test_side_overlay.jpg"
)

print("Landmarks:", result["landmarks"])
print("JSON:", result["landmarks_json"])
print("Overlay saved at:", result["overlay_path"])