.PHONY: setup
setup: .installed

# This file gets touched when setup.sh succeeds.
.installed:
	./setup.sh

# Rather than working out the right dependencies, just rebuild this every time.
# If a human is rebuilding it, it's probably because something has changed.
.PHONY: dist/main
dist/archive.tar.gz: .installed src/*
	./build.sh

test:
	PYTHONPATH=./src pytest

# Following checks disabled:
# C0114: modules don't have docstrings
# C0115: classes don't have docstrings
# E1101: linter doesn't know that cv2 has methods
lint:
	pylint --disable=C0114,C0115,E1101 src/
