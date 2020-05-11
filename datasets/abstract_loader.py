#!/usr/bin/env python3
# encoding: utf-8

"""
    Copyright (C) 2020  Andreas Kuster

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Andreas Kuster"
__copyright__ = "Copyright 2020"
__license__ = "GPL"

from abc import ABC, abstractmethod


class AbstractLoader(ABC):

    def __init__(self):
        """
        Initialize data loader.
        """
        pass

    @abstractmethod
    def load_data(self, text_map_func=lambda x: x, tag_map_func=lambda x: x):
        """
        Load unmodified data, with an optional custom mapping from the text and/or tag.
        :param text_map_func: text map function (default: identity)
        :param tag_map_func: tag map function (default: identity)
        :return: Tuple[List, List] containing the text and tag lists
        """
        raise NotImplementedError()

    @abstractmethod
    def load_data_lowercase(self):
        """
        Load lower cased data.
        :return: Tuple[List, List] containing the text and tag lists
        """
        raise NotImplementedError()

    @abstractmethod
    def load_data_truecase(self):
        """
        Load truecased data.
        :return: Tuple[List, List] containing the text and tag lists
        """
        raise NotImplementedError()

    @abstractmethod
    def load_data_cased_and_uncased(self):
        """
        Load cased and uncased data (2x dataset size)
        :return: Tuple[List, List] containing the text and tag lists
        """
        raise NotImplementedError()

    @abstractmethod
    def load_data_half_mixed(self):
        """
        Load cased and uncased data (50% randomly lowercased)
        :return: Tuple[List, List] containing the text and tag lists
        """
        raise NotImplementedError()
