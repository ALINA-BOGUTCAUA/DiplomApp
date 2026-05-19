package com.example.diplomapp.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface CheckHistoryDao {
    @Insert
    suspend fun insert(checkHistory: CheckHistory): Long

    @Query("SELECT * FROM check_history ORDER BY timestamp DESC")
    fun getAllChecks(): Flow<List<CheckHistory>>

    @Query("SELECT * FROM check_history ORDER BY timestamp DESC LIMIT :limit")
    fun getRecentChecks(limit: Int): Flow<List<CheckHistory>>

    @Query("SELECT * FROM check_history WHERE id = :id")
    suspend fun getCheckById(id: Long): CheckHistory?

    @Query("DELETE FROM check_history WHERE id = :id")
    suspend fun deleteCheck(id: Long)

    @Query("DELETE FROM check_history")
    suspend fun clearHistory()

    @Query("SELECT COUNT(*) FROM check_history WHERE isVoiceFake = 1")
    fun getScamCount(): Flow<Int>
}