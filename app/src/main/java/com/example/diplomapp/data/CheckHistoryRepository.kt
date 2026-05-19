package com.example.diplomapp.data

import kotlinx.coroutines.flow.Flow

class CheckHistoryRepository(private val checkHistoryDao: CheckHistoryDao) {
    val allChecks: Flow<List<CheckHistory>> = checkHistoryDao.getAllChecks()
    val recentChecks: Flow<List<CheckHistory>> = checkHistoryDao.getRecentChecks(20)
    val scamCount: Flow<Int> = checkHistoryDao.getScamCount()

    suspend fun insert(checkHistory: CheckHistory): Long {
        return checkHistoryDao.insert(checkHistory)
    }

    suspend fun getCheckById(id: Long): CheckHistory? {
        return checkHistoryDao.getCheckById(id)
    }

    suspend fun deleteCheck(id: Long) {
        checkHistoryDao.deleteCheck(id)
    }

    suspend fun clearHistory() {
        checkHistoryDao.clearHistory()
    }
}