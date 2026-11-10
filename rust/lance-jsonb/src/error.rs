// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use core::fmt::Display;

use serde::de;
use serde::ser;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParseErrorCode {
    InvalidEOF,
    InvalidNumberValue,
    InvalidStringValue,
    ExpectedSomeIdent,
    ExpectedSomeValue,
    ExpectedColon,
    ExpectedArrayCommaOrEnd,
    ExpectedObjectCommaOrEnd,
    UnexpectedTrailingCharacters,
    KeyMustBeAString,
    ControlCharacterWhileParsingString,
    InvalidEscaped(u8),
    InvalidHex(u8),
    InvalidLoneLeadingSurrogateInHexEscape(u16),
    InvalidSurrogateInHexEscape(u16),
    UnexpectedEndOfHexEscape,
    ObjectDuplicateKey(String),
    ObjectKeyInvalidNumber,
    ObjectKeyInvalidCharacter,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Display for ParseErrorCode {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Self::InvalidEOF => f.write_str("EOF while parsing a value"),
            Self::InvalidNumberValue => f.write_str("invalid number"),
            Self::InvalidStringValue => f.write_str("invalid string"),
            Self::ExpectedSomeIdent => f.write_str("expected ident"),
            Self::ExpectedSomeValue => f.write_str("expected value"),
            Self::ExpectedColon => f.write_str("expected `:`"),
            Self::ExpectedArrayCommaOrEnd => f.write_str("expected `,` or `]`"),
            Self::ExpectedObjectCommaOrEnd => f.write_str("expected `,` or `}`"),
            Self::UnexpectedTrailingCharacters => f.write_str("trailing characters"),
            Self::KeyMustBeAString => f.write_str("key must be a string"),
            Self::ControlCharacterWhileParsingString => {
                f.write_str("control character (\\u0000-\\u001F) found while parsing a string")
            }
            Self::InvalidEscaped(n) => {
                write!(f, "invalid escaped '{:X}'", n)
            }
            Self::InvalidHex(n) => {
                write!(f, "invalid hex '{:X}'", n)
            }
            Self::InvalidLoneLeadingSurrogateInHexEscape(n) => {
                write!(f, "lone leading surrogate in hex escape '{:X}'", n)
            }
            Self::InvalidSurrogateInHexEscape(n) => {
                write!(f, "invalid surrogate in hex escape '{:X}'", n)
            }
            Self::UnexpectedEndOfHexEscape => f.write_str("unexpected end of hex escape"),
            Self::ObjectDuplicateKey(key) => {
                write!(f, "duplicate object attribute \"{}\"", key)
            }
            Self::ObjectKeyInvalidNumber => f.write_str("object attribute name cannot be a number"),
            Self::ObjectKeyInvalidCharacter => {
                f.write_str("object attribute name cannot be invalid character")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    InvalidUtf8,
    InvalidEOF,
    InvalidToken,
    InvalidCast,

    InvalidJson,
    InvalidJsonb,
    InvalidJsonbHeader,
    InvalidJsonbJEntry,
    InvalidJsonbNumber,
    InvalidJsonbExtension,

    InvalidJsonPath,
    InvalidJsonPathPredicate,
    InvalidKeyPath,

    InvalidJsonType,
    InvalidObject,
    ObjectDuplicateKey,
    UnexpectedType,

    Message(String),
    Syntax(ParseErrorCode, usize),
}

impl ser::Error for Error {
    fn custom<T: Display>(msg: T) -> Self {
        Self::Message(msg.to_string())
    }
}

impl de::Error for Error {
    fn custom<T: Display>(msg: T) -> Self {
        Self::Message(msg.to_string())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Message(m) => write!(f, "{}", m),
            Self::Syntax(code, pos) => write!(f, "{}, pos {}", code, pos),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for Error {
    fn from(_error: std::io::Error) -> Self {
        Self::InvalidUtf8
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(_error: std::str::Utf8Error) -> Self {
        Self::InvalidUtf8
    }
}

impl From<nom::Err<nom::error::Error<&[u8]>>> for Error {
    fn from(_error: nom::Err<nom::error::Error<&[u8]>>) -> Self {
        Self::InvalidJsonb
    }
}
