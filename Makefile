.PHONY: docs
experiments_image = acs:4

test:
	py.test

notebook:
	jupyter lab --notebook-dir .

publish_experiments_docker_image:
	docker build -f Dockerfile -t $(experiments_image) .
	docker tag $(experiments_image) khozzy/$(experiments_image)
	docker push khozzy/$(experiments_image)

execute_notebooks:
#	papermill notebooks/rACS_Corridor.ipynb docs/source/notebooks/rACS_Corridor.ipynb
	papermill notebooks/FrozenLake.ipynb docs/source/notebooks/FrozenLake.ipynb
	papermill notebooks/Maze.ipynb docs/source/notebooks/Maze.ipynb
	papermill notebooks/ACS2.ipynb docs/source/notebooks/ACS2.ipynb
