"""Compare model predictions and route using the quantized student."""
import time
import gradio as gr
import yaml
from src.models.common import ROOT
from src.models.inference import NAMES, Predictor, load_thresholds
from src.nlp.normalize import normalize
from src.nlp.extract import extract_model_code
from src.nlp.edit_distance import correct_products
from src.nlp.eliza import reply

# Loading at startup avoids including model-loading time in each prediction.
thresholds = load_thresholds()
models = {name: Predictor(name) for name in NAMES}
teams = yaml.safe_load((ROOT / 'configs/teams.yaml').read_text())['teams']


def route(text):
    normalized = normalize(text)
    code = extract_model_code(normalized)
    corrected = correct_products(normalized)
    response = reply(corrected)
    if response:
        return response, code or 'None', [], 'Small talk'
    predictions = []
    final = 'out_of_scope'
    for name, model in models.items():
        start = time.perf_counter()
        label, confidence = model.predict(corrected, thresholds[name])
        elapsed = (time.perf_counter() - start) * 1000
        predictions.append([name, label, round(confidence, 4), round(elapsed, 2)])
        if name == 'student_int8':
            final = label
    team = 'human agent'
    if final != 'out_of_scope':
        prefix = final.split('_')[0]
        team = next((team for team, prefixes in teams.items() if final in prefixes or prefix in prefixes), 'human agent')
    return corrected, code or 'None', predictions, team


with gr.Blocks(title='Customer Support Ticket Router') as demo:
    gr.Markdown('# Customer Support Ticket Router')
    message = gr.Textbox(label='Message / Mesaj')
    submit = gr.Button('Route / Yönlendir')
    processed = gr.Textbox(label='Processed message or reply')
    code = gr.Textbox(label='Model code')
    table = gr.Dataframe(headers=['Model', 'Intent', 'Confidence', 'Latency (ms)'], interactive=False)
    team = gr.Textbox(label='Final route')
    submit.click(route, inputs=message, outputs=[processed, code, table, team])

if __name__ == '__main__':
    demo.launch()
