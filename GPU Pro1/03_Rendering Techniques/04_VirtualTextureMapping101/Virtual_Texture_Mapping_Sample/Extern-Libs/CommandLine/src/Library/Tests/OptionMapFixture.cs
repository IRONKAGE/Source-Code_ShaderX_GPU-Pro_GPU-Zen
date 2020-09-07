#region Copyright (C) 2005 - 2008 Giacomo Stelluti Scala
//
// Command Line Library: OptionMapFixture.cs
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
        using NUnit.Framework;

        [TestFixture]
        public class OptionMapFixture
        {

                private static OptionMap optionMap;

                [SetUp]
                public void CreateInstance()
                {
                        optionMap = new OptionMap(3);
                }

                [TearDown]
                public void ShutdownOptionMap()
                {
                        optionMap = null;
                }

                [Test]
                public void ManageOptions()
                {
                        OptionAttribute attribute1 = new OptionAttribute("p", "pretend");
                        OptionAttribute attribute2 = new OptionAttribute(null, "newuse");
                        OptionAttribute attribute3 = new OptionAttribute("D", null);

                        OptionInfo option1 = attribute1.CreateOptionInfo();
                        OptionInfo option2 = attribute2.CreateOptionInfo();
                        OptionInfo option3 = attribute3.CreateOptionInfo();

                        optionMap[attribute1.UniqueName] = option1;
                        optionMap[attribute2.UniqueName] = option2;
                        optionMap[attribute3.UniqueName] = option3;

                        Assert.AreSame(option1, optionMap[attribute1.UniqueName]);
                        Assert.AreSame(option2, optionMap[attribute2.UniqueName]);
                        Assert.AreSame(option3, optionMap[attribute3.UniqueName]);
                }
        }
}
#endif