#ifndef OFFLINE_RECORDER_H
#define OFFLINE_RECORDER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef CONFIG_OMI_OFFLINE_RECORDER

/**
 * @brief File header written at the start of each recording segment.
 *
 * Layout (16 bytes):
 *   [0..3]  magic      0x4F4D4952 ("OMIR" = Omi Recording)
 *   [4..7]  version    header format version (currently 1)
 *   [8..11] epoch_s    UTC epoch seconds when recording started
 *   [12..15] reserved  zero-padded, future use
 */
#define OFFLINE_REC_MAGIC   0x4F4D4952
#define OFFLINE_REC_VERSION 1
#define OFFLINE_REC_HEADER_SIZE 16

/**
 * @brief Periodic timestamp marker injected into the audio stream.
 *
 * Stored as a special frame with length prefix = 0xFF (impossible for
 * real Opus frames which max at ~80 bytes), followed by 4-byte epoch.
 */
#define OFFLINE_REC_TIMESTAMP_MARKER 0xFF
#define OFFLINE_REC_TIMESTAMP_SIZE   5  /* 1 byte marker + 4 bytes epoch */

/* Interval between timestamp markers in the audio stream (seconds) */
#define OFFLINE_REC_TIMESTAMP_INTERVAL_S 60

/**
 * @brief Initialize the offline recorder subsystem.
 *
 * Writes a file header with the current RTC timestamp to the SD card.
 * Should be called once after SD card and RTC are initialized.
 *
 * @return 0 on success, negative error code on failure.
 */
int offline_recorder_init(void);

/**
 * @brief Write a periodic timestamp marker into the audio stream.
 *
 * Called by the pusher thread periodically. The marker is written
 * directly to the SD card as a special frame so the app can
 * reconstruct exact timing during playback/stitching.
 */
void offline_recorder_write_timestamp_if_due(void);

/**
 * @brief Get the epoch time when the current recording started.
 */
uint32_t offline_recorder_get_start_epoch(void);

#endif /* CONFIG_OMI_OFFLINE_RECORDER */
#endif /* OFFLINE_RECORDER_H */
