package com.example.diplomapp.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "check_history")
data class CheckHistory(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val fileName: String,
    val filePath: String,
    val modelUsed: String,
    val checkType: String,
    val result: String,
    val confidence: Float,
    val isVoiceFake: Boolean,
    val timestamp: Long = System.currentTimeMillis()
)