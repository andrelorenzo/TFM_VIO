#pragma once

bool runtimeStopRequested();
void runtimeRequestStop();
void runtimeResetControl();

bool runtimeIsPaused();
void runtimeSetPaused(bool paused);
void runtimeTogglePaused();

// Returns true if a global stop was requested while waiting.
bool runtimeWaitIfPaused();
