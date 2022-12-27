#!/bin/bash

delete_cmd="rm -rf __pycache__ hyperparameters logs models nas prophets saved_plots"
arguments=`echo "$@" | tr '[:upper:]' '[:upper:]'`
if [[ "$arguments" =~ "--force" ]] ; then
  eval "$delete_cmd"
else
  while  true; do
      read -p "Do you want to delete all files generated by stock-pred-v3, the Prophet (except datasets)?" yn
      case $yn in
        [Yy]* ) eval "$delete_cmd" ; exit ;;
        [Nn]* ) exit ;;
        * ) echo "This is a simple Y/n question!" ;;
      esac
  done
fi

