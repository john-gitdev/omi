import 'dart:async';

import 'package:flutter/material.dart';

import 'package:omi/backend/schema/bt_device/bt_device.dart';
import 'package:omi/services/services.dart';

/// Lightweight device state for offline mode.
/// No backend calls, no analytics, no Firebase.
class OfflineDeviceProvider extends ChangeNotifier {
  BtDevice? connectedDevice;
  int batteryLevel = -1;
  bool isReconnecting = false;
  StreamSubscription<List<int>>? _batteryListener;

  void setDevice(BtDevice device) {
    connectedDevice = device;
    isReconnecting = false;
    notifyListeners();
    startBatteryListener(device.id);
  }

  void setReconnecting(bool value) {
    isReconnecting = value;
    notifyListeners();
  }

  void clearDevice() {
    connectedDevice = null;
    batteryLevel = -1;
    _batteryListener?.cancel();
    _batteryListener = null;
    notifyListeners();
  }

  Future<void> startBatteryListener(String deviceId) async {
    _batteryListener?.cancel();
    try {
      var connection = await ServiceManager.instance().device.ensureConnection(deviceId);
      if (connection != null) {
        _batteryListener = connection.getBleBatteryLevelListener(
          onBatteryLevelChange: (int level) {
            batteryLevel = level;
            notifyListeners();
          },
        );
      }
    } catch (_) {}
  }

  @override
  void dispose() {
    _batteryListener?.cancel();
    super.dispose();
  }
}
