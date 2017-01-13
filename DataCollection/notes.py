import logging

logger = logging.getLogger(__name__)
logger.warning("test")
print(full_frame.groupby('APROG_PROG_STATUS').count(), full_frame.shape,
      full_frame.groupby('APROG_PROG_STATUS').count() / full_frame.shape[0])

prog_status_CM = len(full_frame[full_frame["APROG_PROG_STATUS"] == 'CM'])
prog_status_DC = len(full_frame[full_frame["APROG_PROG_STATUS"] == 'CM'])