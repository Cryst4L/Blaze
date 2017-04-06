////////////////////////////////////////////////////////////////////////////////
// Timer class using the system absolute (wall) clock.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <sys/time.h>

class Timer {
  private :
	struct timeval t0, t1;

  public :
	Timer() { reset(); }

	void reset() { gettimeofday(&t0, NULL); }

	float getMillisec()
	{
		gettimeofday(&t1, NULL);
		float time = (t1.tv_sec  - t0.tv_sec ) * 1e3
		           + (t1.tv_usec - t0.tv_usec) / 1e3;
		return time;
	}

	~Timer() {}
};
