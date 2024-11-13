FROM python:3.9.16-slim AS builder-image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.7.1 \
    NQS_PATH="/nqs"
ENV VENV_PATH="${NQS_PATH}/.venv/" \
    POETRY_HOME="${NQS_PATH}/.poetry"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# force build from this line on
# ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache

# cache apt packages
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential gcc curl && \
    echo "========== Clean up ==========" && \
    apt-get remove -y && \
    apt-get autoremove -y && \
    rm -rf /tmp/*

# docker user
ARG DOCKER_USER
ARG DOCKER_UID
ARG DOCKER_GID
ENV DOCKER_GROUP=$DOCKER_USER \
    DOCKER_HOME=/home/$DOCKER_USER
RUN groupadd -g $DOCKER_GID $DOCKER_GROUP && \
    useradd -m -u $DOCKER_UID -g $DOCKER_GID -G sudo $DOCKER_USER && \
    echo "$DOCKER_USER:"`openssl rand -base64 12` | chpasswd && \
    echo "$DOCKER_USER ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN mkdir -p /workspace && chown -R ${DOCKER_USER}:${DOCKER_GROUP} /workspace
RUN mkdir -p ${NQS_PATH} && chown -R ${DOCKER_USER}:${DOCKER_GROUP} ${NQS_PATH}

USER ${DOCKER_USER}
RUN echo "python -m venv ${VENV_PATH}" >> ${DOCKER_HOME}/.bashrc && \
    echo "source ${VENV_PATH}/bin/activate" >> ${DOCKER_HOME}/.bashrc

# cache python dependencies only
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/dev/pyproject.toml /packages/dev/
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/tests/pyproject.toml /packages/tests/
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/core/pyproject.toml /packages/core/
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/llm/pyproject.toml /packages/llm/
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/app/pyproject.toml packages/app/poetry.lock /packages/app/
COPY --chown=${DOCKER_USER}:${DOCKER_USER} docker/cp_void_package.sh /packages/
RUN --mount=type=cache,target=${DOCKER_HOME}/.cache,uid=${DOCKER_UID},gid=${DOCKER_GID},sharing=locked \
    /bin/bash -c "python -m venv ${VENV_PATH} && source ${VENV_PATH}/bin/activate && \
    /bin/bash /packages/cp_void_package.sh && cd /tmp/packages/app && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    poetry install" && \
    echo "========== Clean up ==========" && \
    rm -rf /tmp/*

# install now our source code
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/dev /packages/dev
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/tests /packages/tests
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/core /packages/core
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/llm /packages/llm
COPY --chown=${DOCKER_USER}:${DOCKER_USER} packages/app /packages/app
WORKDIR /packages/app
RUN /bin/bash -c "python -m venv ${VENV_PATH} && source ${VENV_PATH}/bin/activate && \
    poetry install --no-cache --sync"

# TODO: for shorter prod images, create a new stage here that copies only what's needed from builder-image

EXPOSE 8501
EXPOSE ${NQS_FLASK_APP_SERVER_PORT}

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/bin/sh", "-c", "streamlit run /packages/app/src/nqs/st_app/Home.py --server.port=8501 --server.address=0.0.0.0"]
