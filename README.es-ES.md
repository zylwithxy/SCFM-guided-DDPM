# Modelo de difusión de eliminación de ruido guiado por flujo de autoajuste para la transferencia de poses humanas
El código fuente de nuestro artículo: ["Self-Calibration Flow Guided Denoising Diffusion Model for Human Pose Transfer"](https://ieeexplore.ieee.org/document/10483084) [Aceptado por TCSVT 2024] \
Yu Xue, Lai-Man Po, Wing-Yin Yu, Haoxuan Wu, Xuyuan Xu, Kun Li, Yuyang Liu


## Resumen

El objetivo del proceso de transferencia de poses humanas es generar imágenes sintéticas de personas que conserven el estilo de las imágenes de referencia y se alineen con precisión con la pose objetivo. Sin embargo, los métodos actuales basados en redes generativas antagónicas (GANs) tienen dificultades para producir detalles realistas y a menudo sufren problemas de desalineación espacial. Por otro lado, los métodos que dependen de modelos de difusión de eliminación de ruido requieren un gran número de parámetros, lo que resulta en tasas de convergencia más lentas. Para abordar estos desafíos, proponemos un módulo guiado por flujo de autoajuste (SCFM) que establece una correspondencia espacial precisa entre las imágenes de referencia y las poses objetivo, facilitando así que el modelo de difusión de eliminación de ruido prediga el ruido de manera más efectiva en cada paso de denoising. Además, introducimos un módulo de fusión de características multiescala (MSFF) que mejora la arquitectura U-Net de eliminación de ruido mediante un mecanismo de atención cruzada, logrando un mejor rendimiento con menos parámetros. Nuestro modelo propuesto supera a los métodos más avanzados en los conjuntos de datos DeepFashion y Market-1501 en términos tanto de cantidad como de calidad de las imágenes sintetizadas.


## Resultados generados
Puedes descargar directamente nuestros resultados de prueba del conjunto de datos DeepFashion desde [Google Drive](https://drive.google.com/file/d/1B850vIDIN7P2PwpLdFwTjIvZtPtFwA8q/view?usp=sharing).


## Conjunto de datos

- Seguimos el mismo método de procesamiento de datos que PIDM (https://github.com/ankanbhunia/PIDM).
- Descarga el archivo `img_highres.zip` del conjunto de datos DeepFashion desde [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00).

- Extrae `img_highres.zip`. Necesitarás pedir la contraseña a los [mantenedores del dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Luego, renombra la carpeta obtenida como **img** y colócala en tu directorio `specified_path`.

- Dividimos el conjunto de entrenamiento y prueba siguiendo el método de GFLA (https://github.com/RenYurui/Global-Flow-Local-Attention). Se eliminaron varias imágenes con oclusiones significativas del conjunto de entrenamiento. Descarga las parejas de entrenamiento/prueba y los puntos clave `pose.zip` extraídos con Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose) manualmente:

  - Descarga las parejas de entrenamiento/prueba desde [Google Drive](https://drive.google.com/drive/folders/1PhnaFNg9zxMZM-ccJAzLIt2iqWFRzXSw?usp=sharing), incluyendo **train_pairs.txt**, **test_pairs.txt**, **train.lst**, **test.lst**. Coloca estos archivos en el directorio `specified_path`.
  - Descarga los puntos clave `pose.rar` extraídos con Openpose desde [Google Drive](https://drive.google.com/file/d/1waNzq-deGBKATXMU9JzMDWdGsF4YkcW_/view?usp=sharing). Extrae y coloca la carpeta obtenida en el directorio `specified_path`.

- Ejecuta el siguiente comando para guardar las imágenes en el conjunto de datos LMDB:

  ```bash
  python data/prepare_data.py \
  --root specified_path \
  --out specified_path
  ```


## Instalación

```bash
# 1. Crea un entorno virtual de conda.
conda create -n SCFM python=3.6
conda activate SCFM
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 2. Clona el repositorio e instala las dependencias
git clone https://github.com/zylwithxy/SCFM-guided-DDPM.git
bash setup.sh # instala las dependencias del módulo de flujo
pip install -r requirements.txt

```


## Entrenamiento

```bash
bash ./scripts/train.sh
```


## Inferencia

El modelo entrenado se encuentra en la carpeta `./checkpoints`.

```bash
bash ./scripts/inference.sh
```


## Cita

Si utilizas los resultados y el código en tu investigación, por favor cita nuestro artículo:

```
@article{xue2024self,
  title={Self-Calibration Flow Guided Denoising Diffusion Model for Human Pose Transfer},
  author={Xue, Yu and Po, Lai-Man and Yu, Wing-Yin and Wu, Haoxuan and Xu, Xuyuan and Li, Kun and Liu, Yuyang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```


### Agradecimientos
Nuestro código se basa en GFLA (https://github.com/RenYurui/Global-Flow-Local-Attention) y PIDM (https://github.com/ankanbhunia/PIDM), a quienes agradecemos por su excelente trabajo.
