#
# Makefile
# Sean Kirmani, 2017-09-17 16:49
#

init::
	@git config core.hooksPath .githooks

all: init

clean: init

# vim:ft=make
#
