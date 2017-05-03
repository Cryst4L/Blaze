#include "../inc/parser.h"

Parser::Parser(int argc, char * argv[])
  :	m_words(argv + 1, argv + argc) {}

bool Parser::checkOption(const char * opt)
{
    std::vector <std::string> ::iterator it;
    it = std::find(m_words.begin(), m_words.end(), opt);
	return (it != m_words.end());
}

const char * Parser::getOption(const char * opt)
{
    std::vector <std::string> ::iterator it;
    it = std::find(m_words.begin(), m_words.end(), opt);

    const char * entry = "";

    if (it != m_words.end() && ++it != m_words.end())
        entry =  it->c_str();

    return entry;
}

Parser::~Parser() {}
