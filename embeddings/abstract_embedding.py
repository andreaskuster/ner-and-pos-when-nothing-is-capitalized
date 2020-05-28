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
from typing import List


class AbstractEmbedding(ABC):

    def __init__(self):
        """
        Initialize embedding class.
        """
        pass

    @abstractmethod
    def word2vec(self, word: str) -> List[float]:
        """
        Converts a given word to its vector representation.
        :param word: input word
        :return: vector representation of the word
        """
        raise NotImplementedError()
