import gradio as gr
import joblib
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw, AllChem


# ============================================================
# Load trained model and fingerprint settings
# ============================================================

MODEL_FILE = "bbbp_random_forest.pkl"
FINGERPRINT_FILE = "fingerprint_info.pkl"

model = joblib.load(MODEL_FILE)
fingerprint_info = joblib.load(FINGERPRINT_FILE)

FP_RADIUS = fingerprint_info.get("radius", 2)
FP_BITS = fingerprint_info.get("n_bits", 2048)


# ============================================================
# Prediction function
# ============================================================

def predict_bbbp(smiles):
    if not smiles or not smiles.strip():
        return (
            "⚠️ Please enter a SMILES string.",
            0.0,
            0.0,
            None
        )

    smiles = smiles.strip()

    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return (
            "❌ Invalid SMILES",
            0.0,
            0.0,
            None
        )

    try:
        # Generate Morgan fingerprint
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=FP_RADIUS,
            nBits=FP_BITS
        )

        # Convert fingerprint to NumPy array
        features = np.zeros((FP_BITS,), dtype=np.int8)
        for i in range(FP_BITS):
            features[i] = fingerprint[i]

        X = features.reshape(1, -1)

        # Prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Find probability belonging to each class
        classes = list(model.classes_)

        if 1 in classes:
            positive_probability = probabilities[classes.index(1)]
        else:
            positive_probability = 0.0

        if 0 in classes:
            negative_probability = probabilities[classes.index(0)]
        else:
            negative_probability = 0.0

        # Prediction label
        if prediction == 1:
            prediction_text = "🟢 BBB-positive"
        else:
            prediction_text = "🔴 BBB-negative"

        # Molecule image
        molecule_image = Draw.MolToImage(
            mol,
            size=(450, 300)
        )

        return (
            prediction_text,
            round(float(negative_probability) * 100, 2),
            round(float(positive_probability) * 100, 2),
            molecule_image
        )

    except Exception as e:
        return (
            f"❌ Prediction error: {str(e)}",
            0.0,
            0.0,
            None
        )


# ============================================================
# Custom CSS
# ============================================================

custom_css = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}

.title {
    text-align: center;
}

.disclaimer {
    padding: 12px;
    border-radius: 10px;
    background: #fff3cd;
    color: #664d03;
}
"""


# ============================================================
# Gradio interface
# ============================================================

with gr.Blocks(
    title="BBBP Predictor",
    css=custom_css,
    theme=gr.themes.Soft()
) as app:

    gr.Markdown(
        """
        # 🧬 BBBP Predictor

        ### Blood-Brain Barrier Penetration Prediction

        Enter a molecule's **SMILES notation** to obtain an
        experimental machine-learning prediction.
        """,
        elem_classes="title"
    )

    with gr.Row():

        with gr.Column():

            smiles_input = gr.Textbox(
                label="🧪 Molecule SMILES",
                placeholder="Example: CCO",
                lines=2
            )

            predict_button = gr.Button(
                "🔬 Predict BBB Penetration",
                variant="primary"
            )

            clear_button = gr.ClearButton(
                components=[smiles_input],
                value="🗑️ Clear"
            )

        with gr.Column():

            prediction_output = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            with gr.Row():

                negative_output = gr.Number(
                    label="🔴 BBB-negative probability (%)",
                    precision=2,
                    interactive=False
                )

                positive_output = gr.Number(
                    label="🟢 BBB-positive probability (%)",
                    precision=2,
                    interactive=False
                )

    molecule_image = gr.Image(
        label="🧬 Molecular Structure",
        type="pil"
    )

    gr.Markdown(
        """
        ---

        ## 📊 Model Information

        **Algorithm:** Random Forest Classifier

        **Features:** Morgan Fingerprints

        **Fingerprint:** Radius 2 · 2048 bits

        **Dataset:** BBBP (Blood-Brain Barrier Penetration)

        **Test Accuracy:** ~89.11%

        **ROC-AUC:** 0.9188

        ---

        ### ⚠️ Scientific Disclaimer

        This is an **experimental machine-learning prediction tool**.
        It should not be used as a substitute for experimental,
        clinical, or medical evidence.

        The predictions are model outputs and do not establish whether
        a molecule will actually cross the human blood-brain barrier.
        """
    )

    predict_button.click(
        fn=predict_bbbp,
        inputs=smiles_input,
        outputs=[
            prediction_output,
            negative_output,
            positive_output,
            molecule_image
        ]
    )


# ============================================================
# Launch
# ============================================================

app.launch()
