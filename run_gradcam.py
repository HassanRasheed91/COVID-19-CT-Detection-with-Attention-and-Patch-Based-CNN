# run_gradcam.py
if __name__ == "__main__":
    from ensemble import PatchAttentionEnsemble
    from gradcam import apply_gradcam
    model = PatchAttentionEnsemble()
    model.load_state_dict(torch.load("model.pth"))
    apply_gradcam(model, "./test/COVID/sample_1.png")
