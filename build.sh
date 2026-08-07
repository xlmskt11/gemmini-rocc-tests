#!/usr/bin/env bash

if [ ! -d "build" ] ; then
    autoconf && \
        mkdir build && cd build && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
elif [ "Makefile.in" -nt "build/Makefile" ] ; then
    # Keep forwarding targets in an existing out-of-tree build synchronized
    # with their canonical Makefile.in definition.
    (cd build && ../configure) || exit 1
fi

cd build

if command -v riscv64-unknown-linux-gnu-gcc >/dev/null 2>&1 || \
   command -v riscv64-linux-gnu-gcc >/dev/null 2>&1 ; then
    make -j "$@"
else
    make -j BAREMETAL_ONLY=1 "$@"
fi
