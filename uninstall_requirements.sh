#!/bin/sh

for plugin in $(cat requirements_no_ver.txt); do
    PLUGIN=$(echo "$plugin" ) # | awk -F == '{print }'
    echo "Uninstalling $PLUGIN..."
    expect -c "spawn pip uninstall $PLUGIN
    expect {
        \"Proceed (y/n)?\" {
            send \"y\r\n\"
            expect {
                exit
            }
        }
    }"
done