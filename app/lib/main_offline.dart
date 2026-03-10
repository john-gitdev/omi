import 'dart:async';

import 'package:flutter/material.dart';

import 'package:flutter_blue_plus/flutter_blue_plus.dart' as ble;
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:opus_dart/opus_dart.dart';
import 'package:opus_flutter/opus_flutter.dart' as opus_flutter;
import 'package:provider/provider.dart';
import 'package:talker_flutter/talker_flutter.dart';

import 'package:omi/backend/preferences.dart';
import 'package:omi/env/env.dart';
import 'package:omi/env/offline_env.dart';
import 'package:omi/flavors.dart';
import 'package:omi/l10n/app_localizations.dart';
import 'package:omi/offline/offline_home_page.dart';
import 'package:omi/providers/connectivity_provider.dart';
import 'package:omi/providers/device_provider.dart';
import 'package:omi/providers/locale_provider.dart';
import 'package:omi/providers/onboarding_provider.dart';
import 'package:omi/services/services.dart';
import 'package:omi/utils/logger.dart';
import 'package:omi/utils/platform/platform_service.dart';

/// Offline-only entry point.
///
/// Run with:
///   flutter run -t lib/main_offline.dart
///
/// This skips Firebase, authentication, analytics, and all backend services.
/// Only BLE device connection, local recording, and device status are available.

Future<void> _initOffline() async {
  // Force offline environment
  F.env = Environment.offline;
  Env.init(OfflineEnv());

  // Service manager (BLE, mic, local services only)
  await ServiceManager.init();

  await SharedPreferencesUtil.init();

  // Mark onboarding as completed for offline mode
  SharedPreferencesUtil().onboardingCompleted = true;

  if (PlatformService.isMobile) initOpus(await opus_flutter.load());

  if (!PlatformService.isWindows) {
    ble.FlutterBluePlus.setOptions(restoreState: true);
    ble.FlutterBluePlus.setLogLevel(ble.LogLevel.info, color: true);
  }

  await ServiceManager.instance().start();
}

void main() {
  runZonedGuarded(
    () async {
      WidgetsFlutterBinding.ensureInitialized();
      await _initOffline();
      runApp(const OfflineApp());
    },
    (error, stack) {
      debugPrint('Uncaught error: $error\n$stack');
    },
  );
}

class OfflineApp extends StatefulWidget {
  const OfflineApp({super.key});

  static final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

  @override
  State<OfflineApp> createState() => _OfflineAppState();
}

class _OfflineAppState extends State<OfflineApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    if (state == AppLifecycleState.detached) {
      ServiceManager.instance().deinit();
    }
  }

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ListenableProvider(create: (context) => ConnectivityProvider()),
        ChangeNotifierProvider(create: (context) => DeviceProvider()),
        ChangeNotifierProxyProvider<DeviceProvider, OnboardingProvider>(
          create: (context) => OnboardingProvider(),
          update: (BuildContext context, value, OnboardingProvider? previous) =>
              (previous?..setDeviceProvider(value)) ?? OnboardingProvider(),
        ),
        ChangeNotifierProvider(create: (context) => LocaleProvider()),
      ],
      builder: (context, child) {
        return MaterialApp(
          debugShowCheckedModeBanner: true,
          title: F.title,
          navigatorKey: OfflineApp.navigatorKey,
          locale: context.watch<LocaleProvider>().locale,
          localizationsDelegates: const [
            AppLocalizations.delegate,
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
          supportedLocales: AppLocalizations.supportedLocales,
          theme: ThemeData(
            useMaterial3: false,
            colorScheme: const ColorScheme.dark(
              primary: Colors.black,
              secondary: Colors.deepPurple,
              surface: Colors.black38,
            ),
            snackBarTheme: const SnackBarThemeData(
              backgroundColor: Color(0xFF1F1F25),
              contentTextStyle: TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.w500),
            ),
            textTheme: TextTheme(
              titleLarge: const TextStyle(fontSize: 18, color: Colors.white),
              titleMedium: const TextStyle(fontSize: 16, color: Colors.white),
              bodyMedium: const TextStyle(fontSize: 14, color: Colors.white),
              labelMedium: TextStyle(fontSize: 12, color: Colors.grey.shade200),
            ),
          ),
          themeMode: ThemeMode.dark,
          builder: (context, child) {
            ErrorWidget.builder = (errorDetails) {
              return Center(
                child: Text(
                  'Error: ${errorDetails.exceptionAsString()}',
                  style: const TextStyle(color: Colors.red),
                ),
              );
            };
            return child!;
          },
          home: TalkerWrapper(
            talker: Logger.instance.talker,
            options: const TalkerWrapperOptions(enableErrorAlerts: false, enableExceptionAlerts: false),
            child: const OfflineHomePage(),
          ),
        );
      },
    );
  }
}
