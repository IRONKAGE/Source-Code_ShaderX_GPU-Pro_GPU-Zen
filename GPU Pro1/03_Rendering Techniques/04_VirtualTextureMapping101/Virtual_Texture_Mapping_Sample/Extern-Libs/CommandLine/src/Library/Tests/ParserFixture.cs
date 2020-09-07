#region Copyright (C) 2005 - 2008 Giacomo Stelluti Scala
//
// Command Line Library: ParserFixture.cs
//
// Author:
//   Giacomo Stelluti Scala (giacomo.stelluti@gmail.com)
//
// Copyright (C) 2005 - 2008 Giacomo Stelluti Scala
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#endregion

#if UNIT_TESTS
namespace CommandLine.Tests
{
        using System;
        using System.Text;
        using System.Collections.Generic;
        using NUnit.Framework;

        [TestFixture]
        public class ParserFixture
        {
                #region Mock Objects
                private class MockOptions
                {
                        [Option("s", "string")]
                        public string StringOption = string.Empty;

                        [Option("i", null)]
                        public int IntOption = 0;

                        [Option(null, "switch")]
                        public bool BoolOption = false;

                        public override string ToString()
                        {
                                StringBuilder builder = new StringBuilder("s/string: ");
                                builder.Append(this.StringOption);
                                builder.Append(Environment.NewLine);
                                builder.Append("i: ");
                                builder.Append(this.IntOption);
                                builder.Append(Environment.NewLine);
                                builder.Append("switch: ");
                                builder.Append(this.BoolOption);
                                builder.Append(Environment.NewLine);
                                return builder.ToString();
                        }
                }

                private class MockBoolPrevalentOptions
                {

                        [Option("a", "option-a")]
                        public bool OptionA = false;

                        [Option("b", "option-b")]
                        public bool OptionB = false;

                        [Option("c", "option-c")]
                        public bool OptionC = false;

                        [Option("d", "double")]
                        public double DoubleOption = 0;

                        public override string ToString()
                        {
                                StringBuilder builder = new StringBuilder("a/option-a: ");
                                builder.Append(this.OptionA);
                                builder.Append(Environment.NewLine);
                                builder.Append("b/option-b: ");
                                builder.Append(this.OptionB);
                                builder.Append(Environment.NewLine);
                                builder.Append("c/option-c: ");
                                builder.Append(this.OptionC);
                                builder.Append(Environment.NewLine);
                                return builder.ToString();
                        }
                }

                private class MockOptionsWithValueList
                {

                        [Option("o", "output")]
                        public string OutputFile = string.Empty;

                        [Option("w", "overwrite")]
                        public bool Overwrite = false;

                        [ValueList(typeof(List<string>))]
                        public IList<string> InputFilenames = null;

                        public override string ToString()
                        {
                                StringBuilder builder = new StringBuilder("o/output: ");
                                builder.Append(this.OutputFile);
                                builder.Append(Environment.NewLine);
                                builder.Append("w/overwrite: ");
                                builder.Append(this.Overwrite);
                                builder.Append(Environment.NewLine);
                                foreach (string inputFile in this.InputFilenames)
                                {
                                        builder.Append("input filename: ");
                                        builder.Append(inputFile);
                                        builder.Append(Environment.NewLine);
                                }
                                return builder.ToString();
                        }
                }

                private class MockOptionsWithValueListMaxElemDefined
                {
                        [Option("o", "output")]
                        public string OutputFile = string.Empty;

                        [Option("w", "overwrite")]
                        public bool Overwrite = false;

                        [ValueList(typeof(List<string>), MaximumElements = 3)]
                        public IList<string> InputFilenames = null;
                }

                private class MockOptionsWithValueListMaxElemEqZero
                {
                        [ValueList(typeof(List<string>), MaximumElements = 0)]
                        public IList<string> Junk = null;
                }

                private class MockOptionsWithOptionList
                {
                        [Option("f", "filename")]
                        public string FileName = string.Empty;

                        [OptionList("s", "search", ':')]
                        public IList<string> SearchKeywords = null;
                }
                #endregion

                [Test]
                [ExpectedException(typeof(ArgumentNullException))]
                public void WillThrowExceptionIfArgumentsArrayIsNull()
                {
                        Parser.ParseArguments(null, new MockOptions());
                }

                [Test]
                [ExpectedException(typeof(ArgumentNullException))]
                public void WillThrowExceptionIfOptionsInstanceIsNull()
                {
                        Parser.ParseArguments(new string[] { }, null);
                }

                [Test]
                [ExpectedException(typeof(ArgumentNullException))]
                public void WillThrowExceptionIfTextWriterIsNull()
                {
                        Parser.ParseArguments(new string[] { }, new MockOptions(), null);
                }

                [Test]
                public void StringOption()
                {
                        MockOptions options = new MockOptions();
                        bool success = Parser.ParseArguments(
                                new string[] { "-s", "something" }, options);

                        Assert.IsTrue(success);
                        Assert.AreEqual("something", options.StringOption);
                        Console.WriteLine(options);
                }

                [Test]
                public void StringIntBoolOptions()
                {
                        MockOptions options = new MockOptions();
                        bool success = Parser.ParseArguments(
                                new string[] { "-s", "another string", "-i100", "--switch" } , options);

                        Assert.IsTrue(success);
                        Assert.AreEqual("another string", options.StringOption);
                        Assert.AreEqual(100, options.IntOption);
                        Assert.AreEqual(true, options.BoolOption);
                        Console.WriteLine(options);
                }

                [Test]
                public void ShortAdjacentOptions()
                {
                        MockBoolPrevalentOptions options = new MockBoolPrevalentOptions();
                        bool success = Parser.ParseArguments(
                                new string[] { "-ca", "-d65" }, options);

                        Assert.IsTrue(success);
                        Assert.IsTrue(options.OptionC);
                        Assert.IsTrue(options.OptionA);
                        Assert.IsFalse(options.OptionB);
                        Assert.AreEqual(65, options.DoubleOption);
                        Console.WriteLine(options);
                }

                [Test]
                public void ShortLongOptions()
                {
                        MockBoolPrevalentOptions options = new MockBoolPrevalentOptions();
                        bool success = Parser.ParseArguments(
                                new string[] { "-b", "--double=9" }, options);

                        Assert.IsTrue(success);
                        Assert.IsTrue(options.OptionB);
                        Assert.IsFalse(options.OptionA);
                        Assert.IsFalse(options.OptionC);
                        Assert.AreEqual(9, options.DoubleOption);
                        Console.WriteLine(options);
                }

                [Test]
                public void ValueListAttributeIsolatesNonOptionValues()
                {
                        MockOptionsWithValueList options = new MockOptionsWithValueList();
                        bool success = Parser.ParseArguments(
                            new string[] { "file1.ext", "file2.ext", "file3.ext", "-wo", "out.ext" }, options);

                        Assert.IsTrue(success);
                        Assert.AreEqual("file1.ext", options.InputFilenames[0]);
                        Assert.AreEqual("file2.ext", options.InputFilenames[1]);
                        Assert.AreEqual("file3.ext", options.InputFilenames[2]);
                        Assert.AreEqual("out.ext", options.OutputFile);
                        Assert.IsTrue(options.Overwrite);
                        Console.WriteLine(options);
                }

                [Test]
                public void ValueListWithMaxElemInsideBounds()
                {
                        MockOptionsWithValueListMaxElemDefined options = new MockOptionsWithValueListMaxElemDefined();
                        bool success = Parser.ParseArguments(
                                new string[] { "file.a", "file.b", "file.c" }, options);

                        Assert.IsTrue(success);
                        Assert.AreEqual("file.a", options.InputFilenames[0]);
                        Assert.AreEqual("file.b", options.InputFilenames[1]);
                        Assert.AreEqual("file.c", options.InputFilenames[2]);
                        Assert.AreEqual(String.Empty, options.OutputFile);
                        Assert.IsFalse(options.Overwrite);
                        Console.WriteLine(options);
                }

                [Test]
                public void ValueListWithMaxElemOutsideBounds()
                {
                        MockOptionsWithValueListMaxElemDefined options = new MockOptionsWithValueListMaxElemDefined();
                        bool success = Parser.ParseArguments(
                                new string[] { "file.a", "file.b", "file.c", "file.d" }, options);

                        Assert.IsFalse(success);
                }

                [Test]
                public void ValueListWithMaxElemSetToZeroSucceeds()
                {
                        MockOptionsWithValueListMaxElemEqZero options = new MockOptionsWithValueListMaxElemEqZero();
                        bool success = Parser.ParseArguments(new string[] { }, options);

                        Assert.IsTrue(success);
                        Assert.AreEqual(0, options.Junk.Count);
                }

                [Test]
                public void ValueListWithMaxElemSetToZeroFailes()
                {
                        MockOptionsWithValueListMaxElemEqZero options = new MockOptionsWithValueListMaxElemEqZero();
                        bool success = Parser.ParseArguments(new string[] { "some", "value" }, options);

                        Assert.IsFalse(success);
                }
                
                [Test]
                public void OptionList()
                {
                        MockOptionsWithOptionList options = new MockOptionsWithOptionList();
                        bool success = Parser.ParseArguments(new string[] {
                                "-s", "string1:stringTwo:stringIII", "-f", "test-file.txt" }, options );

                        Assert.IsTrue(success);
                        Assert.AreEqual("string1", options.SearchKeywords[0]);
                        Console.WriteLine(options.SearchKeywords[0]);
                        Assert.AreEqual("stringTwo", options.SearchKeywords[1]);
                        Console.WriteLine(options.SearchKeywords[1]);
                        Assert.AreEqual("stringIII", options.SearchKeywords[2]);
                        Console.WriteLine(options.SearchKeywords[2]);
                }

                /// <summary>
                /// Ref.: #BUG0000.
                /// </summary>
                [Test]
                public void ShortOptionRefusesEqualToken()
                {
                        MockOptions options = new MockOptions();

                        Assert.IsFalse(Parser.ParseArguments(new string[] { "-i=10" }, options));
                        Console.WriteLine(options);
                }

                /// <summary>
                /// Ref.: #BUG0001
                /// </summary>
                [Test]
                [ExpectedException(typeof(MissingMethodException))]
                public void CanNotCreateParserInstance()
                {
                        Activator.CreateInstance(typeof(Parser));
                }
        }
}
#endif
