package com.example.diplomapp.data

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit

class SettingsRepository(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    var isCallMonitoringEnabled: Boolean
        get() = prefs.getBoolean(KEY_CALL_MONITORING, false)
        set(value) = prefs.edit { putBoolean(KEY_CALL_MONITORING, value) }

    var defaultCheckType: String
        get() = prefs.getString(KEY_DEFAULT_CHECK_TYPE, CHECK_TYPE_FAST) ?: CHECK_TYPE_FAST
        set(value) = prefs.edit { putString(KEY_DEFAULT_CHECK_TYPE, value) }

    var defaultModel: String
        get() = prefs.getString(KEY_DEFAULT_MODEL, "CNN") ?: "CNN"
        set(value) = prefs.edit { putString(KEY_DEFAULT_MODEL, value) }

    var autoCheckOnCall: Boolean
        get() = prefs.getBoolean(KEY_AUTO_CHECK_ON_CALL, true)
        set(value) = prefs.edit { putBoolean(KEY_AUTO_CHECK_ON_CALL, value) }

    companion object {
        private const val PREFS_NAME = "voice_check_prefs"
        private const val KEY_CALL_MONITORING = "call_monitoring_enabled"
        private const val KEY_DEFAULT_CHECK_TYPE = "default_check_type"
        private const val KEY_DEFAULT_MODEL = "default_model"
        private const val KEY_AUTO_CHECK_ON_CALL = "auto_check_on_call"

        const val CHECK_TYPE_FAST = "fast"
        const val CHECK_TYPE_FULL = "full"
    }
}