#region Copyright (C) 2005 - 2008 Giacomo Stelluti Scala
//
// Command Line Library: OptionMap.cs
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

namespace CommandLine
{
        using System;
        using System.Collections.Generic;
        using System.Collections.Specialized;

        internal sealed class OptionMap : Dictionary<string, OptionInfo>
        {
                private StringDictionary names;

                public OptionMap(int capacity)
                        : base(capacity)
                {
                        this.names = new StringDictionary();
                }

                new public OptionInfo this[string key]
                {
                        get
                        {
                                OptionInfo option = null;
                                if (base.ContainsKey(key))
                                {
                                        option = base[key];
                                }
                                else
                                {
                                        string optionKey = names[key];
                                        if (optionKey != null)
                                        {
                                                option = base[optionKey];
                                        }
                                }
                                return option;
                        }
                        set
                        {
                                base[key] = value;
                                if (value.HasBothNames)
                                {
                                        names[value.LongName] = value.ShortName;
                                }
                        }
                }

                public bool EnforceRules()
                {
                        foreach (OptionInfo option in this.Values)
                        {
                                if (option.Required && !option.IsDefined)
                                {
                                        return false;
                                }
                        }
                        return true;
                }
        }
}
