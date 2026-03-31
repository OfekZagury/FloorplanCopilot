FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system libs required by matplotlib/opencv at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline_code/requirements.azure.txt /tmp/requirements.azure.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.azure.txt

# Keep package layout so imports like `ResPlan_RAG.pipeline_code...` work.
COPY . /app/ResPlan_RAG

EXPOSE 80

ENV GRADIO_HOST=0.0.0.0 \
    GRADIO_PORT=80 \
    WEBSITES_PORT=80 \
    ENABLE_POST_DONE_RENDER=false \
    ENABLE_WEB_EDITOR=true

CMD ["python", "-m", "ResPlan_RAG.pipeline_code.gradio_GUI.run_web_app"]

