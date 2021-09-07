
import fastbook
#fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
from fastai.imports import *

path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')

btn_upload = widgets.FileUpload()
btn_upload

img = PILImage.create(btn_upload.data[-1])

out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl

pred,pred_idx,probs = learn_inf.predict(img)

lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred

btn_run = widgets.Button(description='Classify')
btn_run

def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)

btn_upload = widgets.FileUpload()

VBox([widgets.Label('Upload a photo of your meal!'),
      btn_upload, btn_run, out_pl, lbl_pred])