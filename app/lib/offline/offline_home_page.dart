import 'package:flutter/material.dart';

import 'package:provider/provider.dart';

import 'package:omi/backend/preferences.dart';
import 'package:omi/backend/schema/bt_device/bt_device.dart';
import 'package:omi/offline/offline_device_provider.dart';
import 'package:omi/providers/onboarding_provider.dart';
import 'package:omi/services/services.dart';
import 'package:omi/widgets/dialog.dart';

class OfflineHomePage extends StatefulWidget {
  const OfflineHomePage({super.key});

  @override
  State<OfflineHomePage> createState() => _OfflineHomePageState();
}

class _OfflineHomePageState extends State<OfflineHomePage> {
  final OfflineDeviceProvider _deviceState = OfflineDeviceProvider();

  @override
  void initState() {
    super.initState();
    _deviceState.addListener(() {
      if (mounted) setState(() {});
    });
    _tryReconnectSavedDevice();
  }

  @override
  void dispose() {
    _deviceState.dispose();
    super.dispose();
  }

  Future<void> _tryReconnectSavedDevice() async {
    final saved = SharedPreferencesUtil().btDevice;
    if (saved.id.isNotEmpty) {
      _deviceState.setReconnecting(true);
      try {
        var connection = await ServiceManager.instance().device.ensureConnection(saved.id);
        if (connection != null) {
          final info = await saved.getDeviceInfo(connection);
          _deviceState.setDevice(info ?? saved);
        } else {
          _deviceState.setReconnecting(false);
        }
      } catch (_) {
        _deviceState.setReconnecting(false);
      }
    }
  }

  void _startScan() {
    final provider = context.read<OnboardingProvider>();
    provider.scanDevices(
      onShowDialog: () {
        if (mounted) {
          showDialog(
            context: context,
            builder: (c) => getDialog(
              context,
              () => Navigator.of(context).pop(),
              () {},
              'Enable Bluetooth',
              'Bluetooth is required to connect to your Omi device.',
              singleButton: true,
            ),
          );
        }
      },
    );
  }

  Future<void> _connectDevice(BtDevice device) async {
    _deviceState.setReconnecting(true);
    try {
      var connection = await ServiceManager.instance().device.ensureConnection(device.id);
      if (connection != null) {
        final info = await device.getDeviceInfo(connection);
        final connected = info ?? device;
        _deviceState.setDevice(connected);
        SharedPreferencesUtil().btDevice = connected;

        // Start battery listener
        _deviceState.startBatteryListener(connected.id);
      } else {
        _deviceState.setReconnecting(false);
      }
    } catch (e) {
      _deviceState.setReconnecting(false);
    }
  }

  void _disconnectDevice() {
    _deviceState.clearDevice();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text('Omi Offline Recorder', style: TextStyle(color: Colors.white)),
        centerTitle: true,
      ),
      body: _deviceState.connectedDevice != null ? _buildConnectedView() : _buildScanView(),
    );
  }

  Widget _buildConnectedView() {
    final device = _deviceState.connectedDevice!;
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Device info card
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFF1A1A2E),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.green.withOpacity(0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 12,
                      height: 12,
                      decoration: const BoxDecoration(color: Colors.green, shape: BoxShape.circle),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        device.name.isNotEmpty ? device.name : 'Omi Device',
                        style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w600),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _infoRow('Firmware', device.firmwareRevision ?? 'Unknown'),
                if (_deviceState.batteryLevel >= 0) _infoRow('Battery', '${_deviceState.batteryLevel}%'),
                _infoRow('Mode', 'Offline Recorder'),
                _infoRow('Storage', 'SD Card'),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Status card
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFF1A1A2E),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Recording Status',
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Icon(Icons.fiber_manual_record, color: Colors.red.shade400, size: 16),
                    const SizedBox(width: 8),
                    const Text(
                      'Recording to SD Card',
                      style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w500),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                const Text(
                  'Audio is being captured and stored locally on the device SD card.',
                  style: TextStyle(color: Colors.white54, fontSize: 13),
                ),
              ],
            ),
          ),
          const Spacer(),

          // Disconnect button
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: _disconnectDevice,
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: Colors.red),
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
              child: const Text('Disconnect', style: TextStyle(color: Colors.red, fontSize: 16)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.white54, fontSize: 14)),
          Text(value, style: const TextStyle(color: Colors.white, fontSize: 14)),
        ],
      ),
    );
  }

  Widget _buildScanView() {
    if (_deviceState.isReconnecting) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: Colors.deepPurple),
            SizedBox(height: 16),
            Text('Connecting...', style: TextStyle(color: Colors.white70, fontSize: 16)),
          ],
        ),
      );
    }

    return Consumer<OnboardingProvider>(
      builder: (context, provider, child) {
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 40),
              const Icon(Icons.bluetooth_searching, color: Colors.deepPurple, size: 64),
              const SizedBox(height: 24),
              const Text(
                'Connect Your Omi Device',
                style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              const Text(
                'The device will record audio to its SD card.\nNo cloud services required.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white54, fontSize: 14),
              ),
              const SizedBox(height: 32),

              // Scan button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _startScan,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.deepPurple,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: Text(
                    provider.deviceList.isEmpty ? 'Scan for Devices' : 'Rescan',
                    style: const TextStyle(fontSize: 16, color: Colors.white),
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // Device list
              Expanded(
                child: ListView.builder(
                  itemCount: provider.deviceList.length,
                  itemBuilder: (context, index) {
                    final device = provider.deviceList[index];
                    return Container(
                      margin: const EdgeInsets.only(bottom: 8),
                      decoration: BoxDecoration(
                        color: const Color(0xFF1A1A2E),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ListTile(
                        leading: const Icon(Icons.bluetooth, color: Colors.deepPurple),
                        title: Text(
                          device.name.isNotEmpty ? device.name : 'Unknown Device',
                          style: const TextStyle(color: Colors.white),
                        ),
                        subtitle: Text(
                          'RSSI: ${device.rssi} dBm',
                          style: const TextStyle(color: Colors.white38, fontSize: 12),
                        ),
                        trailing: const Icon(Icons.arrow_forward_ios, color: Colors.white38, size: 16),
                        onTap: () => _connectDevice(device),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
