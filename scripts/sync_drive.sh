exit 1

rclone sync -v --progress -L /lustre/groups/ml01/workspace/kemal.inecik/tardis_data gdrive:/tardis/tardis_data

rclone sync -v --progress /Users/kemalinecik/git_nosync/tardis/data gdrive:/tardis/tardis_data \
--exclude "/raw/**" \
--exclude "__pycache__/**" \
--exclude "*.DS_Store" \
--exclude ".idea/**" \
--exclude ".mypy_cache/**" \
--exclude "*.ipynb_checkpoints/**" \
--exclude ".git/**" 

rclone sync -v --progress /Users/kemalinecik/git_nosync/tardis gdrive:/tardis/repo \
--exclude "__pycache__/**" \
--exclude "*.DS_Store" \
--exclude ".idea/**" \
--exclude ".mypy_cache/**" \
--exclude "*.ipynb_checkpoints/**" \
--exclude ".git/**" \
--exclude "/data/**"

rclone sync -v --progress gdrive:/tardis/tardis_data /Users/kemalinecik/git_nosync/tardis/data --exclude "/raw/**"

rclone sync -v --progress gdrive:/tardis/tardis_data /lustre/groups/ml01/workspace/kemal.inecik/tardis_data --exclude "/raw/**"


