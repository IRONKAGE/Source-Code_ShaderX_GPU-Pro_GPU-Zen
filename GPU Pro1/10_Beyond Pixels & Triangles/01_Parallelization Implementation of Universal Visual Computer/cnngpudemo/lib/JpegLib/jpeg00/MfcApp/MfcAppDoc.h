// MfcAppDoc.h : interface of the CMfcAppDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_MFCAPPDOC_H__7B4DD98D_945B_11D2_815A_444553540000__INCLUDED_)
#define AFX_MFCAPPDOC_H__7B4DD98D_945B_11D2_815A_444553540000__INCLUDED_

#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000


class CMfcAppDoc : public CDocument
{
protected: // create from serialization only
	CMfcAppDoc();
	DECLARE_DYNCREATE(CMfcAppDoc)

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CMfcAppDoc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CMfcAppDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CMfcAppDoc)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Developer Studio will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_MFCAPPDOC_H__7B4DD98D_945B_11D2_815A_444553540000__INCLUDED_)
