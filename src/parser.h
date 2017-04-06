#pragma once

#include <string>
#include <algorithm>
#include <iostream>

class Parser
{
  public:

    Parser(int argc, char * argv[])
    :   m_words(argv + 1, argv + argc) {}

    bool checkOption(const char * opt)
    {
        std::vector <std::string> ::iterator it;
        it = std::find(m_words.begin(), m_words.end(), opt);
    	return (it != m_words.end());
    }

    const char * getOption(const char * opt)
    {
        std::vector <std::string> ::iterator it;
        it = std::find(m_words.begin(), m_words.end(), opt);

        const char * entry = "";

        if (it != m_words.end() && ++it != m_words.end())
            entry =  it->c_str();

        return entry;
    }

    ~Parser() {}

  private:

    std::vector <std::string> m_words;
};
