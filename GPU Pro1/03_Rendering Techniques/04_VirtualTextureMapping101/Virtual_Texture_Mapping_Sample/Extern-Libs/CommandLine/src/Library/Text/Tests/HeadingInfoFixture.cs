#region Copyright (C) 2005 - 2008 Giacomo Stelluti Scala
//
// Command Line Library: HeadingInfoFixture.cs
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
namespace CommandLine.Text.Tests
{
        using System;
        using System.IO;
        using NUnit.Framework;

        [TestFixture]
        public class HeadingInfoFixture
        {
                [Test]
                public void OnlyProgramName()
                {
                        HeadingInfo hi = new HeadingInfo("myprog");
                        string s = hi;
                        Assert.AreEqual("myprog", s);
                        StringWriter sw = new StringWriter();
                        hi.WriteMessage("a message", sw);
                        Assert.AreEqual("myprog: a message" + Environment.NewLine, sw.ToString());
                }

                [Test]
                public void ProgramNameAndVersion()
                {
                        HeadingInfo hi = new HeadingInfo("myecho", "2.5");
                        string s = hi;
                        Assert.AreEqual("myecho 2.5", s);
                        StringWriter sw = new StringWriter();
                        hi.WriteMessage("hello unit-test", sw);
                        Assert.AreEqual("myecho: hello unit-test" + Environment.NewLine, sw.ToString());
                }
        }
}
#endif
