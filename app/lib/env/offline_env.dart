import 'env.dart';

/// Environment for offline-only builds.
/// All keys are null — no backend, analytics, or third-party services.
final class OfflineEnv implements EnvFields {
  OfflineEnv();

  @override
  final String? openAIAPIKey = null;

  @override
  final String? mixpanelProjectToken = null;

  @override
  final String? apiBaseUrl = null;

  @override
  final String? growthbookApiKey = null;

  @override
  final String? googleMapsApiKey = null;

  @override
  final String? intercomAppId = null;

  @override
  final String? intercomIOSApiKey = null;

  @override
  final String? intercomAndroidApiKey = null;

  @override
  final String? googleClientId = null;

  @override
  final String? googleClientSecret = null;

  @override
  final bool? useWebAuth = false;

  @override
  final bool? useAuthCustomToken = false;

  @override
  final String? stagingApiUrl = null;
}
