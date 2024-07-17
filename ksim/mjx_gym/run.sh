export MODEL_DIR=$HOME/ksim/ksim # This assumes you cloned the main ksim repo in $HOME; adjust if this is not the case
source activate pytorch
export DISPLAY=:0
export MUJOCO_GL=egl
Xvfb :0 -screen 0 1024x768x24 &
python play.py --config experiments/stompy_walk.yaml --use_mujoco --twitch_stream_key= # Put your stream key here