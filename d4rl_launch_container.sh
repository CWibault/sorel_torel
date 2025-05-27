gpu=$1
WANDB_API_KEY=$(cat ./d4rl_dev/wandb_key)

docker run \
    --gpus device=$gpu \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/sorel_torel \
    --name d4rl_torel\_$gpu \
    --user $(id -u) \
    -it d4rl_torel bash