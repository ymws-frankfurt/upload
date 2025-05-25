#FROM rapidsai/rapidsai-core:23.12-cuda12.7-runtime-ubuntu22.04-py3.10
#FROM rapidsai/rapidsai-core:23.10-cuda12.2-runtime-ubuntu22.04-py3.10
FROM rapidsai/rapidsai-core:cuda11.8-runtime-ubuntu22.04-py3.10

WORKDIR /app/

COPY . .

# Install system dependencies
RUN apt update && \
    apt install -y libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Enable Mamba and set Conda priority
RUN conda install -y mamba -c conda-forge && \
    conda config --set channel_priority strict && \
    rm -rf /opt/conda/pkgs/cache && \
    conda clean --all -y

# Install dependencies with Mamba -> ??? polars[plot]
RUN mamba install -y flask numpy numba scipy scikit-learn pandas -c conda-forge && \
    mamba install -y matplotlib seaborn tqdm statsmodels plotly -c conda-forge && \
    mamba install -y fredapi finta pandas_market_calendars pandas-ta -c conda-forge && \
    conda clean --all -y

# ta ABANDONED!!!(20250227) -> mumba docker compose super fast only 1 minute
# Install remaining packages with pip -> [pykalman icecream] shifted here because while these do not cause error in image building, they are not installed in fact
# plotly kaleido -> reluctant to use but ki33elev visualization useful -> if problem surfaces take another approach!
RUN pip install --no-cache-dir exchange_calendars fmpsdk cvxpy pykalman icecream plotly kaleido duckdb

CMD ["/bin/bash"]
