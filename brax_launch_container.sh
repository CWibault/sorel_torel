gpu=$1
WANDB_API_KEY=$(cat ./brax_dev/wandb_key)

docker run \
    --gpus device=$gpu \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/sorel_torel \
    --name sorel_torel\_$gpu \
    --user $(id -u) \
    -it sorel_torel bash