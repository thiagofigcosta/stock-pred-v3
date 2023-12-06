FROM python:3.9-slim AS slim
RUN printf '#!/bin/bash\n\ntail -f /dev/null\n' > /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]