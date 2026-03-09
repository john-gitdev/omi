#include "offline_recorder.h"

#ifdef CONFIG_OMI_OFFLINE_RECORDER

#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "sd_card.h"
#include "rtc.h"

LOG_MODULE_REGISTER(offline_recorder, CONFIG_LOG_DEFAULT_LEVEL);

static uint32_t recording_start_epoch;
static int64_t last_timestamp_uptime_ms;

int offline_recorder_init(void)
{
    uint32_t now = get_utc_time();
    recording_start_epoch = now;
    last_timestamp_uptime_ms = k_uptime_get();

    /* Write 16-byte file header */
    uint8_t header[OFFLINE_REC_HEADER_SIZE];
    uint32_t magic = OFFLINE_REC_MAGIC;
    uint32_t version = OFFLINE_REC_VERSION;
    uint32_t reserved = 0;

    memcpy(header + 0, &magic, 4);
    memcpy(header + 4, &version, 4);
    memcpy(header + 8, &now, 4);
    memcpy(header + 12, &reserved, 4);

    uint32_t written = write_to_file(header, OFFLINE_REC_HEADER_SIZE);
    if (written != OFFLINE_REC_HEADER_SIZE) {
        LOG_ERR("Failed to write offline recording header");
        return -EIO;
    }

    if (now == 0) {
        LOG_WRN("Offline recorder started without valid RTC time");
    } else {
        LOG_INF("Offline recorder started, epoch=%u", now);
    }

    return 0;
}

void offline_recorder_write_timestamp_if_due(void)
{
    int64_t now_ms = k_uptime_get();
    int64_t elapsed_ms = now_ms - last_timestamp_uptime_ms;

    if (elapsed_ms < (OFFLINE_REC_TIMESTAMP_INTERVAL_S * 1000)) {
        return;
    }

    last_timestamp_uptime_ms = now_ms;

    uint32_t epoch = get_utc_time();
    uint8_t marker[OFFLINE_REC_TIMESTAMP_SIZE];
    marker[0] = OFFLINE_REC_TIMESTAMP_MARKER;
    memcpy(marker + 1, &epoch, 4);

    write_to_file(marker, OFFLINE_REC_TIMESTAMP_SIZE);
    LOG_DBG("Timestamp marker written: epoch=%u", epoch);
}

uint32_t offline_recorder_get_start_epoch(void)
{
    return recording_start_epoch;
}

#endif /* CONFIG_OMI_OFFLINE_RECORDER */
