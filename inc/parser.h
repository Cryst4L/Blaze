////////////////////////////////////////////////////////////////////////////////
// Minimal Argument Parser - B.Halimi 2017
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>
#include <algorithm>
#include <iostream>

class Parser
{
  public:

	Parser(int argc, char * argv[]);

	bool checkOption(const char * opt);

	const char * getOption(const char * opt);

	~Parser();

  private:

	std::vector <std::string> m_words;
};
