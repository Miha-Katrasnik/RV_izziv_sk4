# RV_izziv_sk4
Seminarska naloga pri predmetu Robotski vid (UL FE)

Primeri videoposnetkov in prikaz delovanja se nahaja na povezavi:
<a href="https://drive.google.com/drive/folders/10Jd7BtdS0cPqYwX1KFjaPRAEqrXxwKll?usp=sharing" target="_blank">Google Drive</a>

Testne videoposnetke naložite v mapo videos. V 9. vrstici datoteke <code>RV_seminarska.py</code> lahko spremenite imena datotek, ki jih želite uporabiti.
```{python}
video_file = ['videos/MVI_6342.MOV', 'videos/MVI_6339.MOV'] #vrstni red: leva roka, desna roka
```

Med delovanjem programa lahko s tipko <b>w</b> začasno ustavite program. Nato lahko prikažete različne slike:
* d - razlika med trenutno in prvo sliko
* t - upragovljena slika
* f - barvna slika
* 0 - prva slika posnetka

S tipko q lahko video predčasno zaključite, lahko pa enostavno zaprete okno.

## Opis programa

Program je namenjen avtomatskemu merjenju časa opravljanja preizkusa z devetimi zatiči.

## Knjižnice

Za uporabo programa so potrebne knjižnice:
* OpenCV (testirano za verzijo 4.5.1)
* NumPy
* Matplotlib
