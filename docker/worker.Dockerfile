ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG QAIRT_DEPENDENCIES_FILE
ARG HARNESS_CONSTRAINTS_FILE
ARG TORCH_VERSION
ARG TORCH_INDEX_URL

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        adb \
        ca-certificates \
        clang \
        flatbuffers-compiler \
        libc++-dev \
        libc++abi-dev \
        libflatbuffers-dev \
        libgl1 \
        libllvm14 \
        libncurses6 \
        "libpython${PYTHON_VERSION}" \
        lsb-release \
        make \
        "python${PYTHON_VERSION}" \
        python3-distutils \
        python3-pip \
        "python${PYTHON_VERSION}-venv" \
        rename \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qairt-agent
COPY ${QAIRT_DEPENDENCIES_FILE} ./docker/requirements.txt
RUN "python${PYTHON_VERSION}" -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
    && /opt/venv/bin/python -m pip install --no-cache-dir \
        --requirement docker/requirements.txt

COPY docker/.generated/qairt-agent-src.zip ./qairt-agent-src.zip
COPY ${HARNESS_CONSTRAINTS_FILE} ./harness/constraints.json
# The target registry travels with the constraints: they name a target by
# reference and cannot resolve it without these entries.
COPY harness/targets/ ./harness/targets/

RUN /opt/venv/bin/python -m pip check \
    && /opt/venv/bin/python -c "import sys; expected=tuple(map(int, '${PYTHON_VERSION}'.split('.'))); assert sys.version_info[:2] == expected"

ENV VIRTUAL_ENV=/opt/venv \
    QAIRT_SDK_ROOT=/opt/qairt \
    QNN_SDK_ROOT=/opt/qairt \
    QAIRT_AGENT_HARNESS_CONSTRAINTS=/opt/qairt-agent/harness/constraints.json \
    PYTHONPATH=/opt/qairt-agent/qairt-agent-src.zip:/opt/qairt/lib/python:/opt/qairt/benchmarks/QNN \
    PATH=/opt/venv/bin:/opt/qairt/bin/x86_64-linux-clang:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/qairt/lib/x86_64-linux-clang \
    PYTHONDONTWRITEBYTECODE=1

CMD ["/opt/venv/bin/python", "-m", "qairt_agent.cli", "--help"]
